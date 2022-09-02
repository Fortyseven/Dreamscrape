<script>
  import {
    Container,
    Row,
    Col,
    Image,
    Card,
    Button,
    ButtonGroup,
    Icon,
  } from "sveltestrap";
  import Frobs from "./frobs/Frobs.svelte";
  import Prompt from "./prompt/Prompt.svelte";
  import ResultPanel from "./results/ResultPanel.svelte";
</script>

<Container fluid>
  <Row width="100%">
    <Col xs={5}>
      <Prompt />
      <br />
      <Col>
        <Row>
          <ButtonGroup vertical>
            <Button title="Generate" color="primary"
              ><Icon name="play-fill" /></Button
            >
            <Button title="Reset" color="secondary"
              ><Icon name="arrow-repeat" /></Button
            >
          </ButtonGroup>
        </Row>
      </Col>
      <br />
      <Card>
        <Frobs />
      </Card>
    </Col>
    <Col>
      <ResultPanel />
    </Col>
  </Row>
</Container>
